from itertools import product

import pytest
from safetensors.torch import load_file, save_file

import steering.feature_paths as feature_paths
import steering.stats_io as stats_io
from scripts.train.conditional_encdec_sft import _resolve_val_feature_dir
from steering.feature_io import save_features
from steering.feature_paths import (
    DatasetPhase,
    FeatureDirectoryMismatchError,
    PoolingMode,
    build_feature_dir,
    parse_feature_dir,
    resolve_output_dir,
    verify_feature_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from tests.steering.test_feature_dir_verification import _make_shard, _make_stats


def _kwargs(tmp_path, *, phase=DatasetPhase.STANDARD_SFT):
    return {
        "root": tmp_path,
        "segment_size": 64,
        "layer": 8,
        "activation_type": ActivationType.residual,
        "pooling": PoolingMode.max_pool,
        "width": SAEWidth.W_16K,
        "model": "gemma-2-2b",
        "dataset_phase": phase,
    }


@pytest.mark.parametrize("ci,no_bos,turn", product((False, True), repeat=3))
def test_build_parse_round_trip_all_leaf_combinations(tmp_path, ci, no_bos, turn):
    mode = SegmentMode.turn_delimited_fixed_tokens if turn else SegmentMode.fixed_tokens
    path = build_feature_dir(
        **_kwargs(tmp_path),
        context_independent_features=ci,
        ignore_bos_features=no_bos,
        segment_mode=mode,
    )
    expected_leaves = [
        leaf
        for leaf, enabled in (
            ("context_independent", ci),
            ("no_bos", no_bos),
            ("turn_delimited", turn),
        )
        if enabled
    ]
    if expected_leaves:
        assert list(path.parts[-len(expected_leaves) :]) == expected_leaves
    components = parse_feature_dir(path)
    assert components.context_independent_features is ci
    assert components.ignore_bos_features is no_bos
    assert components.segment_mode is mode


def test_misplaced_turn_leaf_raises(tmp_path):
    canonical = build_feature_dir(
        **_kwargs(tmp_path),
        ignore_bos_features=True,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    misplaced = canonical.parent.parent / "turn_delimited" / "no_bos"
    with pytest.raises(ValueError, match="invalid position"):
        parse_feature_dir(misplaced)


def test_resolve_output_dir_turn_passthrough_and_explicit_untouched(tmp_path):
    auto = resolve_output_dir(
        output_dir=None,
        feature_root=str(tmp_path),
        segment_size=64,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase="standard_sft",
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    assert auto.name == "turn_delimited"
    explicit = tmp_path / "custom"
    assert (
        resolve_output_dir(
            output_dir=str(explicit),
            feature_root="ignored",
            segment_size=64,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase="standard_sft",
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )
        == explicit
    )


@pytest.mark.parametrize("ci,no_bos,turn", product((False, True), repeat=3))
def test_resolve_val_feature_dir_preserves_all_leaves(tmp_path, ci, no_bos, turn):
    mode = SegmentMode.turn_delimited_fixed_tokens if turn else SegmentMode.fixed_tokens
    train = build_feature_dir(
        **_kwargs(tmp_path),
        context_independent_features=ci,
        ignore_bos_features=no_bos,
        segment_mode=mode,
    )
    resolved = parse_feature_dir(
        _resolve_val_feature_dir(None, str(train), "standard", "standard")
    )
    assert resolved.dataset_phase is DatasetPhase.STANDARD_SFT_VAL
    assert resolved.context_independent_features is ci
    assert resolved.ignore_bos_features is no_bos
    assert resolved.segment_mode is mode
    # A steered train phase resolves onto STANDARD's val leaf (val family),
    # carrying every PR-2 leaf unchanged.
    steered_train = build_feature_dir(
        **_kwargs(tmp_path, phase=DatasetPhase.STANDARD_STEERED_SFT),
        context_independent_features=ci,
        ignore_bos_features=no_bos,
        segment_mode=mode,
    )
    steered_val = parse_feature_dir(
        _resolve_val_feature_dir(None, str(steered_train), "standard_steered", "standard")
    )
    assert steered_val.dataset_phase is DatasetPhase.STANDARD_SFT_VAL
    assert steered_val.context_independent_features is ci
    assert steered_val.ignore_bos_features is no_bos
    assert steered_val.segment_mode is mode


def _feature_dir(tmp_path, *, leaf_mode, shard_mode="turn_delimited_fixed_tokens", stats_mode=None):
    path = build_feature_dir(**_kwargs(tmp_path), segment_mode=leaf_mode)
    path.mkdir(parents=True)
    shard = _make_shard(segment_mode=shard_mode)
    if shard_mode is None:
        shard.pop("segment_mode")
    save_features(shard, path / "shard_00000_features")
    stats_mode = stats_mode or leaf_mode.value
    save_statistics(
        _make_stats(segment_mode=stats_mode),
        path / "feature_statistics",
    )
    return path


def test_verify_turn_shard_under_leaf_passes(tmp_path):
    path = _feature_dir(
        tmp_path,
        leaf_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    verify_feature_dir(path)


@pytest.mark.parametrize(
    "leaf_mode,shard_mode,match",
    [
        (
            SegmentMode.turn_delimited_fixed_tokens,
            "fixed_tokens",
            r"Shard .*: segment_mode='fixed_tokens', expected "
            r"'turn_delimited_fixed_tokens' from path",
        ),
        (
            SegmentMode.fixed_tokens,
            "turn_delimited_fixed_tokens",
            r"Shard .*: segment_mode='turn_delimited_fixed_tokens', expected "
            r"'fixed_tokens' from path",
        ),
        (
            SegmentMode.turn_delimited_fixed_tokens,
            None,
            r"Shard .*: shard predates segment_mode metadata but the directory "
            r"carries the turn_delimited leaf",
        ),
    ],
)
def test_verify_shard_mode_mismatches_fail(tmp_path, leaf_mode, shard_mode, match):
    path = _feature_dir(tmp_path, leaf_mode=leaf_mode, shard_mode=shard_mode)
    with pytest.raises(FeatureDirectoryMismatchError, match=match):
        verify_feature_dir(path)


def test_verify_legacy_absent_mode_without_leaf_passes(tmp_path):
    path = _feature_dir(
        tmp_path,
        leaf_mode=SegmentMode.fixed_tokens,
        shard_mode=None,
    )
    verify_feature_dir(path)


def test_verify_stats_mode_mismatch_fails(tmp_path):
    path = _feature_dir(
        tmp_path,
        leaf_mode=SegmentMode.turn_delimited_fixed_tokens,
        stats_mode="fixed_tokens",
    )
    with pytest.raises(FeatureDirectoryMismatchError, match="Statistics: segment_mode"):
        verify_feature_dir(path)


def test_verify_unknown_shard_mode_names_artifact_and_values(tmp_path):
    path = _feature_dir(
        tmp_path,
        leaf_mode=SegmentMode.turn_delimited_fixed_tokens,
        shard_mode="bogus",
    )
    with pytest.raises(
        FeatureDirectoryMismatchError,
        match=r"Shard .* has unknown segment_mode='bogus'; must be one of "
        r"\['fixed_tokens', 'turn_delimited_fixed_tokens'\]",
    ):
        verify_feature_dir(path)


@pytest.mark.parametrize(
    "leaf_mode,should_raise",
    [
        (SegmentMode.turn_delimited_fixed_tokens, True),
        (SegmentMode.fixed_tokens, False),
    ],
)
def test_verify_legacy_statistics_without_mode(tmp_path, leaf_mode, should_raise):
    path = _feature_dir(tmp_path, leaf_mode=leaf_mode, shard_mode=leaf_mode.value)
    stats_path = path / "feature_statistics.safetensors"
    tensors = load_file(stats_path)
    metadata = stats_io._read_metadata(stats_path)
    metadata.pop("segment_mode")
    save_file(tensors, stats_path, metadata=metadata)
    if should_raise:
        with pytest.raises(
            FeatureDirectoryMismatchError,
            match="Statistics: statistics predates segment_mode metadata",
        ):
            verify_feature_dir(path)
    else:
        verify_feature_dir(path)


def test_verify_turn_directory_without_statistics_reports_only_missing_file(tmp_path):
    path = build_feature_dir(
        **_kwargs(tmp_path),
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    path.mkdir(parents=True)
    save_features(
        _make_shard(segment_mode=SegmentMode.turn_delimited_fixed_tokens.value),
        path / "shard_00000_features",
    )
    with pytest.raises(FeatureDirectoryMismatchError) as exc_info:
        verify_feature_dir(path)
    statistics_errors = [
        line for line in str(exc_info.value).splitlines() if "Statistics" in line
    ]
    assert statistics_errors == [f"  - Statistics file not found in {path}"]


def test_feature_paths_and_stats_io_import_together():
    assert feature_paths.SegmentMode is stats_io.SegmentMode
