"""The cluster contract must take a chained SFT checkpoint's feature directory from its online-SAE lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.conditional.encoder_decoder.sft_grpo_online_sae import ONLINE_SAE_LINEAGE_KEY
from steering.eval.cluster_contract_identity import (
    ClusterFeatureContractError,
    FeatureDirectoryIdentity,
)
from steering.eval.cluster_feature_contract import (
    _assert_checkpoint_feature_directory_identity,
    resolve_cluster_feature_contract,
)
from steering.feature_paths import parse_feature_dir
from tests.steering.test_cluster_feature_contract import _write_full_preflight_fixture

_CANONICAL = (
    "/orcd/data/example/features/s32/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/"
    "standard_steered_sft/context_independent/no_bos/turn_delimited"
)
_STAGED_FLAT = "/scratch/henryac/sft-stage/features/s32/train"


def _configured() -> FeatureDirectoryIdentity:
    return FeatureDirectoryIdentity.from_components(parse_feature_dir(_CANONICAL))


def test_lineage_feature_dir_is_preferred_over_flat_staged_user_config() -> None:
    metadata = {
        "user_config": {"feature_dir": _STAGED_FLAT},
        ONLINE_SAE_LINEAGE_KEY: {"feature_dir": _CANONICAL},
    }
    _assert_checkpoint_feature_directory_identity(metadata, _configured())


def test_without_lineage_the_flat_staged_user_config_still_fails_loud() -> None:
    metadata = {"user_config": {"feature_dir": _STAGED_FLAT}}
    with pytest.raises(ClusterFeatureContractError, match="not a canonical feature directory"):
        _assert_checkpoint_feature_directory_identity(metadata, _configured())


def test_pretrain_checkpoint_user_config_feature_dir_is_unchanged() -> None:
    metadata = {"user_config": {"feature_dir": _CANONICAL}}
    _assert_checkpoint_feature_directory_identity(metadata, _configured())


@pytest.mark.parametrize("lineage", [None, [], {}, {"feature_dir": None}, {"feature_dir": 123}])
def test_present_malformed_lineage_cannot_fall_back_to_user_config(lineage: object) -> None:
    metadata = {
        "user_config": {"feature_dir": _CANONICAL},
        ONLINE_SAE_LINEAGE_KEY: lineage,
    }
    with pytest.raises(ClusterFeatureContractError, match="lineage"):
        _assert_checkpoint_feature_directory_identity(metadata, _configured())


@pytest.mark.parametrize("wrong_family", [False, True])
def test_full_cluster_preflight_checks_lineage_directory_for_staged_checkpoint(
    tmp_path: Path, wrong_family: bool,
) -> None:
    request, fake_sae = _write_full_preflight_fixture(tmp_path)
    assert request.checkpoint_dir is not None
    path = Path(request.checkpoint_dir) / "meta_000007.json"
    metadata = json.loads(path.read_text())
    canonical = metadata["user_config"]["feature_dir"]
    metadata["user_config"]["feature_dir"] = _STAGED_FLAT
    metadata[ONLINE_SAE_LINEAGE_KEY] = {
        "feature_dir": canonical.replace("/l8/", "/l9/") if wrong_family else canonical,
    }
    path.write_text(json.dumps(metadata))
    if wrong_family:
        with pytest.raises(ClusterFeatureContractError, match="does not match checkpoint training"):
            resolve_cluster_feature_contract(request, sae_loader=lambda _resolved: fake_sae)
    else:
        contract = resolve_cluster_feature_contract(request, sae_loader=lambda _resolved: fake_sae)
        assert contract.feature_directory == FeatureDirectoryIdentity.from_components(
            parse_feature_dir(canonical)
        )
