"""Contract tests for the pretrain- and SFT-leaf-aware chat setup loader."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from loguru import logger

from scripts.chat_tui import setup as chat_setup
from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract,
    resolve_checkpoint_feature_contract,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.sampling import SamplingBundle, SamplingMode
from steering.eval.axbench_upstream.methods import MethodSetup, load_method_setup
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import N_SAE_FEATURES, build_tiny_model

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "chat_tui"
_STATS_RELATIVE = Path(
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos/"
    "feature_statistics.safetensors"
)


def _find_real_statistics() -> Path:
    worktree_root = Path(__file__).resolve().parents[2]
    candidates = [worktree_root / _STATS_RELATIVE]
    candidates.extend(parent / _STATS_RELATIVE for parent in worktree_root.parents)
    return next((path for path in candidates if path.is_file()), candidates[0])


REAL_STATISTICS = _find_real_statistics()
requires_real_statistics = pytest.mark.skipif(
    not REAL_STATISTICS.is_file(),
    reason=f"real feature statistics are absent: {REAL_STATISTICS}",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_meta(name: str) -> dict[str, Any]:
    loaded = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _nested_lineage(
    tmp_path: Path,
    *,
    leaf_fixture: str = "meta_000749.json",
) -> tuple[Path, Path]:
    ancestor_dir = tmp_path / "pretrain_checkpoint"
    leaf_dir = ancestor_dir / "standard_sft_conditioned_checkpoints"
    leaf_dir.mkdir(parents=True)
    ancestor_path = ancestor_dir / "meta_026500.json"
    leaf_path = leaf_dir / "meta_000749.json"
    shutil.copyfile(FIXTURES / "meta_026500.json", ancestor_path)
    shutil.copyfile(FIXTURES / leaf_fixture, leaf_path)
    return leaf_path, ancestor_path


def _assert_structurally_equal(actual: Any, expected: Any, *, owner: str) -> None:
    assert type(actual) is type(expected), owner
    if isinstance(actual, t.Tensor):
        assert t.equal(actual, expected), owner
        return
    if isinstance(actual, t.Generator):
        assert actual.device == expected.device, owner
        assert t.equal(actual.get_state(), expected.get_state()), owner
        return
    if is_dataclass(actual) and not isinstance(actual, type):
        for field in fields(actual):
            _assert_structurally_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
                owner=f"{owner}.{field.name}",
            )
        return
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), owner
        for key in actual:
            _assert_structurally_equal(actual[key], expected[key], owner=f"{owner}[{key!r}]")
        return
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), owner
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_structurally_equal(actual_item, expected_item, owner=f"{owner}[{index}]")
        return
    assert actual == expected, owner


def test_chat_setup_rejects_unmeasured_turn_checkpoint_before_statistics_or_ancestor_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint" / "meta_000001.json"
    target.parent.mkdir()
    _write_json(
        target,
        {
            "user_config": {},
            "sft_segment_mode": "turn_delimited_fixed_tokens",
        },
    )
    monkeypatch.setattr(
        chat_setup,
        "load_method_setup",
        lambda *_args, **_kwargs: pytest.fail("upstream setup reached"),
    )

    with pytest.raises(ValueError, match="turn_self_cond_segment_size_rustbpe"):
        chat_setup.load_chat_setup(target, tmp_path / "missing-statistics.safetensors")


@requires_real_statistics
def test_pretrain_shape_delegates_field_for_field() -> None:
    target = FIXTURES / "meta_026500.json"

    actual = chat_setup.load_chat_setup(target, REAL_STATISTICS)
    expected = load_method_setup(target, REAL_STATISTICS)

    assert {field.name for field in fields(MethodSetup)} == set(actual.__dict__)
    for field in fields(MethodSetup):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if field.name == "resolved_sae":
            attributes = ("release", "sae_id", "hook_name")
            assert tuple(getattr(actual_value, name) for name in attributes) == tuple(
                getattr(expected_value, name) for name in attributes
            )
        else:
            _assert_structurally_equal(
                actual_value, expected_value, owner=f"MethodSetup.{field.name}"
            )


@requires_real_statistics
def test_sft_leaf_happy_path_resolves_ancestor_identity_and_prompt_contract(
    tmp_path: Path,
) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        setup = chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)
    finally:
        logger.remove(sink)

    assert setup.target_meta_path == leaf_path
    assert setup.meta["step"] == 749
    assert setup.sae_model == "gemma-3-4b-pt"
    assert setup.sae_layer == 22
    assert setup.n_sae_features == 65_536
    assert setup.resolved_sae.sae_id == "layer_22_width_65k_l0_medium"
    assert setup.feature_vocab is not None
    assert setup.feature_vocab.size == 2_074
    assert setup.allowed_ids is not None
    assert setup.allowed_ids.shape == (2_074,)
    assert setup.prompt_feature_contract.feature_mask == setup.feature_contract.feature_mask
    assert (
        setup.prompt_feature_contract.resolved_clusters.definition
        == setup.feature_contract.resolved_clusters.definition
    )

    assert setup.output_selection.mode is SamplingMode.top_k_percentile
    assert setup.output_selection.delta_weights is None
    assert int(setup.output_selection.min_k) == 1
    assert int(setup.output_selection.max_k) == 49
    assert int(setup.output_negative_selection.min_k) == 0
    assert int(setup.output_negative_selection.max_k) == 49

    pipeline = setup.prompt_pipeline
    assert isinstance(pipeline.selection_config, SamplingBundle)
    assert pipeline.selection_config.selection.mode is SamplingMode.top_k_percentile
    assert pipeline.selection_config.selection.delta_weights is None
    assert int(pipeline.selection_config.selection.max_k) == 49
    assert pipeline.negative_bundle is not None
    assert int(pipeline.negative_bundle.selection.min_k) == 0
    assert pipeline.effective_buckets == 1
    assert pipeline.negative_bundle.rng is pipeline.selection_config.rng
    expected_rng = t.Generator(device="cpu").manual_seed(42)
    assert t.equal(pipeline.selection_config.rng.get_state(), expected_rng.get_state())
    assert setup.percentile_table.shape == (65_536, 99)
    assert t.equal(pipeline.selection_config.percentile_table, setup.percentile_table)

    mismatch_warnings = [message for message in messages if "leaf user_config.sae" in message]
    assert len(mismatch_warnings) == 1
    assert "gemma-2-2b/16k" in mismatch_warnings[0]
    assert "gemma-3-4b-pt/65k" in mismatch_warnings[0]


@requires_real_statistics
def test_sft_leaf_prompt_pipeline_uses_leaf_selection_not_ancestor(
    tmp_path: Path,
) -> None:
    leaf_path, ancestor_path = _nested_lineage(
        tmp_path,
        leaf_fixture="meta_000749_distinct_selection.json",
    )

    setup = chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)
    ancestor = json.loads(ancestor_path.read_text(encoding="utf-8"))

    assert isinstance(setup.prompt_pipeline.selection_config, SamplingBundle)
    assert setup.prompt_pipeline.selection_config.selection.mean == 7.0
    assert ancestor["user_config"]["conditioning"]["feature_selection"]["mean"] == 12.0


@requires_real_statistics
def test_corrupted_ancestor_statistics_sha_fails(tmp_path: Path) -> None:
    leaf_path, ancestor_path = _nested_lineage(tmp_path)
    ancestor = _fixture_meta("meta_026500.json")
    expected = ancestor["user_config"]["conditioning"]["stats_source_sha256"]
    ancestor["user_config"]["conditioning"]["stats_source_sha256"] = (
        "0" if expected[0] != "0" else "1"
    ) + expected[1:]
    _write_json(ancestor_path, ancestor)

    with pytest.raises(ValueError, match="do not match the target checkpoint"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_leaf_without_pretrain_conditioning_fails(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    del leaf["pretrain_conditioning"]
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="pretrain_conditioning"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_malformed_highest_step_nearest_ancestor_names_path(tmp_path: Path) -> None:
    leaf_path, ancestor_path = _nested_lineage(tmp_path)
    malformed_path = ancestor_path.parent / "meta_999999.json"
    malformed_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(str(malformed_path))):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_tampered_leaf_feature_vocab_sha_fails_canonical_check(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    leaf["feature_vocab"]["sha256"] = "0" * 64
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="does not match its canonical"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_tampered_leaf_canonical_mask_fails_recorded_vocab_check(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    n_sae_features = int(leaf["feature_vocab"]["n_sae_features"])
    original_contract = resolve_checkpoint_feature_contract(
        leaf,
        n_sae_features=PositiveInt.parse(n_sae_features),
        surface="test leaf",
        source_path=str(leaf_path),
    )
    original_vocab = effective_feature_vocab(original_contract.universe, n_sae_features)
    assert original_vocab is not None
    retained_id = int(original_vocab.original_ids[0])
    canonical_ids = leaf["feature_mask_canonical"]["ids"]
    assert retained_id in canonical_ids
    leaf["feature_mask_canonical"]["ids"] = [
        feature_id for feature_id in canonical_ids if feature_id != retained_id
    ]
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="feature vocabulary does not match its canonical"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


@requires_real_statistics
def test_leaf_feature_mask_path_marker_is_advisory(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    original_marker = leaf["user_config"]["feature_mask"]
    leaf["user_config"]["feature_mask"] = {
        **original_marker,
        "path": "advisory/moved-feature-mask.txt",
    }
    _write_json(leaf_path, leaf)

    setup = chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)

    assert setup.prompt_feature_contract.feature_mask == setup.feature_contract.feature_mask


def test_tampered_leaf_feature_vocab_width_names_leaf_and_ancestor(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    leaf["feature_vocab"]["n_sae_features"] = 16_384
    _write_json(leaf_path, leaf)

    with pytest.raises(
        ValueError,
        match=r"ancestor SAE n_sae_features=65536.*leaf feature_vocab.n_sae_features=16384",
    ):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_leaf_without_negative_selection_fails(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    del leaf["pretrain_conditioning"]["negative_selection"]
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="negative_selection"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_leaf_with_three_buckets_fails(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    leaf["num_buckets"] = 3
    leaf["user_config"]["num_buckets"] = 3
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="two buckets"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_leaf_with_non_top_k_selection_mode_fails(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    leaf["user_config"]["feature_selection"]["mode"] = "weighted"
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="mode"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


@requires_real_statistics
def test_leaf_non_bce_feature_prediction_config_fails(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    leaf["feature_prediction_config"]["loss_fn"] = "mse"
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="loss_fn"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


@requires_real_statistics
def test_leaf_non_max_pool_feature_prediction_config_fails(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf = _fixture_meta("meta_000749.json")
    leaf["feature_prediction_config"]["loss_mode"] = "last_token"
    _write_json(leaf_path, leaf)

    with pytest.raises(ValueError, match="loss_mode"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_leaf_without_any_ancestor_conditioning_meta_fails(tmp_path: Path) -> None:
    leaf_dir = tmp_path / "isolated" / "standard_sft_conditioned_checkpoints"
    leaf_dir.mkdir(parents=True)
    leaf_path = leaf_dir / "meta_000749.json"
    shutil.copyfile(FIXTURES / "meta_000749.json", leaf_path)

    with pytest.raises(ValueError, match="no ancestor checkpoint metadata"):
        chat_setup.load_chat_setup(leaf_path, REAL_STATISTICS)


def test_leaf_conditioning_without_stats_sha_names_key_and_path(tmp_path: Path) -> None:
    leaf_path = tmp_path / "meta_000007.json"
    setup = SimpleNamespace(
        meta={"user_config": {"conditioning": {"seed": 42}}},
        target_meta_path=leaf_path,
    )

    with pytest.raises(ValueError) as exc_info:
        chat_setup.resolve_stats_sha(setup)

    assert str(exc_info.value) == (
        f"leaf conditioning block is present but lacks stats_source_sha256: {leaf_path}"
    )


def test_leaf_conditioning_without_seed_names_key_and_path(tmp_path: Path) -> None:
    leaf_path = tmp_path / "meta_000007.json"
    setup = SimpleNamespace(
        meta={"user_config": {"conditioning": {"stats_source_sha256": "0" * 64}}},
        target_meta_path=leaf_path,
    )

    with pytest.raises(ValueError) as exc_info:
        chat_setup.resolve_conditioning_seed(setup)

    assert str(exc_info.value) == (
        f"leaf conditioning block is present but lacks seed: {leaf_path}"
    )


def _valid_fp_config() -> FeaturePredictionConfig:
    return FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_fn=FeaturePredLossFn.BCE,
        loss_mode=FeaturePredLossMode.MAX_POOL,
    )


def _model_setup(model: Any, *, n_sae_features: int = int(N_SAE_FEATURES)) -> tuple[Any, Any]:
    contract = CheckpointFeatureContract(feature_mask=None, resolved_clusters=None)
    setup = SimpleNamespace(
        meta={"step": 7},
        target_meta_path=Path("checkpoint/meta_000007.json"),
        feature_contract=contract,
        feature_vocab=None,
        n_sae_features=n_sae_features,
    )
    return setup, object()


def _patch_model_load(
    monkeypatch: pytest.MonkeyPatch,
    setup: Any,
    model: Any,
    tokenizer: Any,
) -> None:
    monkeypatch.setattr(
        chat_setup,
        "build_encdec_model",
        lambda *_args, **_kwargs: (model, tokenizer, {"loaded": True}),
    )
    monkeypatch.setattr(
        chat_setup,
        "resolve_and_assert_model_feature_contract",
        lambda *_args, **_kwargs: setup.feature_contract,
    )


def test_load_chat_target_model_returns_axbench_model_tokenizer_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, tokenizer = _model_setup(model)
    _patch_model_load(monkeypatch, setup, model, tokenizer)

    assert chat_setup.load_chat_target_model(setup, "cpu") == (model, tokenizer)


def test_load_chat_target_model_uses_bfloat16_for_cuda_shell_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, tokenizer = _model_setup(model)
    build_default_dtypes: list[t.dtype] = []
    empty_cache_calls: list[None] = []

    def fake_build(*_args: Any, **_kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        build_default_dtypes.append(t.get_default_dtype())
        return model, tokenizer, {"loaded": True}

    monkeypatch.setattr(chat_setup, "build_encdec_model", fake_build)
    monkeypatch.setattr(
        chat_setup.t.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(None),
    )
    monkeypatch.setattr(
        chat_setup,
        "resolve_and_assert_model_feature_contract",
        lambda *_args, **_kwargs: setup.feature_contract,
    )
    previous_default = t.get_default_dtype()

    loaded, loaded_tokenizer = chat_setup.load_chat_target_model(
        setup,
        "cuda",
        dtype="bfloat16",
    )

    assert loaded is model
    assert loaded_tokenizer is tokenizer
    assert build_default_dtypes == [t.bfloat16]
    assert empty_cache_calls == [None]
    assert t.get_default_dtype() is previous_default
    assert {parameter.dtype for parameter in model.parameters()} == {t.bfloat16}


def test_load_chat_target_model_restores_default_dtype_when_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, _tokenizer = _model_setup(model)
    previous_default = t.get_default_dtype()

    def fail_build(*_args: Any, **_kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        assert t.get_default_dtype() is t.bfloat16
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(chat_setup, "build_encdec_model", fail_build)

    with pytest.raises(RuntimeError, match="synthetic load failure"):
        chat_setup.load_chat_target_model(setup, "cuda", dtype="bfloat16")

    assert t.get_default_dtype() is previous_default


def test_load_chat_target_model_rejects_unsupported_cuda_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, _tokenizer = _model_setup(model)
    build_calls: list[None] = []
    monkeypatch.setattr(
        chat_setup,
        "build_encdec_model",
        lambda *_args, **_kwargs: build_calls.append(None),
    )

    with pytest.raises(ValueError, match="bfloat16, float32"):
        chat_setup.load_chat_target_model(setup, "cuda", dtype="float16")

    assert build_calls == []


def test_load_chat_target_model_rejects_wrong_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        feature_vocab=None,
        feature_pred_config=SimpleNamespace(
            loss_fn=FeaturePredLossFn.BCE,
            loss_mode=FeaturePredLossMode.MAX_POOL,
        ),
        feature_pred_head=SimpleNamespace(out_features=int(N_SAE_FEATURES)),
        encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2)),
    )
    setup, tokenizer = _model_setup(model)
    _patch_model_load(monkeypatch, setup, model, tokenizer)

    with pytest.raises(ValueError, match="EncoderDecoderGPT.*SimpleNamespace"):
        chat_setup.load_chat_target_model(setup, "cpu")


@pytest.mark.parametrize(
    "feature_pred_config",
    [
        None,
        SimpleNamespace(
            loss_fn=FeaturePredLossFn.MSE,
            loss_mode=FeaturePredLossMode.MAX_POOL,
        ),
        SimpleNamespace(
            loss_fn=FeaturePredLossFn.BCE,
            loss_mode=FeaturePredLossMode.BEST_TOKEN,
        ),
    ],
)
def test_load_chat_target_model_rejects_incompatible_feature_prediction_config(
    monkeypatch: pytest.MonkeyPatch,
    feature_pred_config: Any,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    model.feature_pred_config = feature_pred_config
    setup, tokenizer = _model_setup(model)
    _patch_model_load(monkeypatch, setup, model, tokenizer)

    with pytest.raises(ValueError, match="feature_prediction"):
        chat_setup.load_chat_target_model(setup, "cpu")


def test_load_chat_target_model_rejects_headless_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    model.feature_pred_head = None
    setup, tokenizer = _model_setup(model)
    _patch_model_load(monkeypatch, setup, model, tokenizer)

    with pytest.raises(ValueError, match="feature_prediction"):
        chat_setup.load_chat_target_model(setup, "cpu")


def test_load_chat_target_model_rejects_three_bucket_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=3,
        feature_pred_config=_valid_fp_config(),
    )
    setup, tokenizer = _model_setup(model)
    _patch_model_load(monkeypatch, setup, model, tokenizer)

    with pytest.raises(ValueError, match="buckets"):
        chat_setup.load_chat_target_model(setup, "cpu")


def test_load_chat_target_model_rejects_prediction_width_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, tokenizer = _model_setup(model, n_sae_features=int(N_SAE_FEATURES) + 1)
    _patch_model_load(monkeypatch, setup, model, tokenizer)

    with pytest.raises(
        ValueError,
        match=rf"head width={int(N_SAE_FEATURES)}.*width={int(N_SAE_FEATURES) + 1}",
    ):
        chat_setup.load_chat_target_model(setup, "cpu")


@pytest.mark.parametrize(
    ("setup_extra", "loaded_extra"),
    [
        ({}, {"turn_self_cond_segment_size_rustbpe": None}),
        (
            {"turn_self_cond_segment_size_rustbpe": 37},
            {"turn_self_cond_segment_size_rustbpe": 37.0},
        ),
    ],
    ids=["absent-vs-null", "int-vs-float"],
)
def test_load_chat_target_model_separates_absent_null_and_numeric_identity_drift(
    monkeypatch: pytest.MonkeyPatch, setup_extra: dict, loaded_extra: dict
) -> None:
    # kills: comparing identity keys with dict.get / raw != (absent == null,
    # 37 == 37.0), the weak form the shared AxBench guard already rejects.
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, tokenizer = _model_setup(model)
    setup.meta.update(setup_extra)
    loaded = {**setup.meta, **loaded_extra}
    monkeypatch.setattr(
        chat_setup,
        "build_encdec_model",
        lambda *_args, **_kwargs: (model, tokenizer, loaded),
    )
    with pytest.raises(ValueError, match="segmentation identity differs"):
        chat_setup.load_chat_target_model(setup, "cpu")


def test_load_chat_target_model_rejects_turn_checkpoint_before_contract_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: dropping the guard on the freshly loaded target metadata, which is
    # the only witness for a turn checkpoint the setup metadata never carried.
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=_valid_fp_config(),
    )
    setup, tokenizer = _model_setup(model)
    monkeypatch.setattr(
        chat_setup,
        "build_encdec_model",
        lambda *_args, **_kwargs: (
            model,
            tokenizer,
            {"sft_segment_mode": "turn_delimited_fixed_tokens"},
        ),
    )
    monkeypatch.setattr(
        chat_setup,
        "resolve_and_assert_model_feature_contract",
        lambda *_args, **_kwargs: pytest.fail("contract resolved for a turn checkpoint"),
    )

    with pytest.raises(ValueError, match="segmentation identity differs"):
        chat_setup.load_chat_target_model(setup, "cpu")
