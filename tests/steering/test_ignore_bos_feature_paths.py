"""ignore_bos_features path build/parse/resolve, all 4 leaf combinations."""

import itertools
from pathlib import Path

import pytest

from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    parse_feature_dir,
    resolve_output_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth

_BASE = {
    "root": "out/data/features",
    "segment_size": 128,
    "layer": 8,
    "activation_type": ActivationType.residual,
    "pooling": PoolingMode.max_pool,
    "width": SAEWidth("16k"),
    "model": "gemma-2-2b",
    "dataset_phase": DatasetPhase.BASE,
}


@pytest.mark.parametrize("ci,ib", list(itertools.product([False, True], repeat=2)))
def test_build_then_parse_roundtrips(ci, ib):
    p = build_feature_dir(
        **_BASE, context_independent_features=ci, ignore_bos_features=ib  # ty: ignore[invalid-argument-type]
    )
    parts = Path(p).parts
    if ib:
        assert parts[-1] == "no_bos"
    if ci:
        assert "context_independent" in parts
    comps = parse_feature_dir(p)
    assert comps.context_independent_features is ci
    assert comps.ignore_bos_features is ib
    assert comps.segment_size == 128
    assert comps.layer == 8
    assert comps.dataset_phase == DatasetPhase.BASE


def test_both_flags_canonical_order():
    p = build_feature_dir(
        **_BASE, context_independent_features=True, ignore_bos_features=True  # ty: ignore[invalid-argument-type]
    )
    assert Path(p).parts[-2:] == ("context_independent", "no_bos")


def test_no_bos_only_leaf():
    p = build_feature_dir(**_BASE, ignore_bos_features=True)  # ty: ignore[invalid-argument-type]
    assert Path(p).name == "no_bos"
    assert Path(p).parts[-2] == "base"


def test_invalid_leaf_order_fails_loudly():
    bad = Path(build_feature_dir(**_BASE)) / "no_bos" / "context_independent"  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="dataset phase"):
        parse_feature_dir(bad)


def test_legacy_path_parses_ignore_bos_false():
    p = build_feature_dir(**_BASE)  # ty: ignore[invalid-argument-type]
    assert parse_feature_dir(p).ignore_bos_features is False


def test_resolve_output_dir_appends_no_bos():
    p = resolve_output_dir(
        output_dir=None,
        feature_root="out/data/features",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth("16k"),
        model="gemma-2-2b",
        dataset_phase="base",
        context_independent_features=False,
        ignore_bos_features=True,
    )
    assert Path(p).name == "no_bos"


def test_parse_too_few_components_message_mentions_leaves():
    """parse_feature_dir on a short path ending in /no_bos raises ValueError with the
    canonical message that names all leaves and reports the stripped-inclusive count."""
    # 5 core components (too few) + /no_bos leaf → stripped=1, reported count=6
    short_with_leaf = "a/b/c/d/e/no_bos"
    with pytest.raises(
        ValueError,
        match="up to 10 with context_independent, no_bos, and/or turn_delimited leaves",
    ):
        parse_feature_dir(short_with_leaf)

    # Also works with both leaves stripped: 5 core + /context_independent/no_bos
    short_with_both = "a/b/c/d/e/context_independent/no_bos"
    with pytest.raises(
        ValueError,
        match="up to 10 with context_independent, no_bos, and/or turn_delimited leaves",
    ):
        parse_feature_dir(short_with_both)


def test_resolve_output_dir_explicit_passthrough_no_leaf():
    p = resolve_output_dir(
        output_dir="/tmp/explicit",
        feature_root="x",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth("16k"),
        model="gemma-2-2b",
        dataset_phase="base",
        context_independent_features=True,
        ignore_bos_features=True,
    )
    assert str(p) == "/tmp/explicit"
