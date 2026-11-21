"""Cluster-contract coverage for feature-GCG checkpoint configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.gcg_config as gcg_config_module
from errors import FeatureVocabError
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureContractError,
)
from tests.steering.feature_gcg.unit.test_clustered_optimizer_contract import (
    _contract_model_pair,
)


def _bucket_call(
    model: Any,
    contract: ClusterFeatureContract,
) -> tuple[int, int]:
    return assert_single_positive_bucket(model, cluster_contract=contract)


def _resolve_call(
    model: Any,
    contract: ClusterFeatureContract,
) -> tuple[int, int]:
    return resolve_k(
        model,
        OursGcgConfig(),
        cluster_contract=contract,
    )


_CONTRACT_CALLS: tuple[
    tuple[str, Callable[[Any, ClusterFeatureContract], tuple[int, int]]],
    ...,
] = (
    ("bucket-layout", _bucket_call),
    ("top-k", _resolve_call),
)


def test_clustered_bucket_layout_happy_path_uses_real_loaded_state(
    tmp_path: Path,
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)

    assert assert_single_positive_bucket(
        model,
        cluster_contract=contract,
    ) == (0, 1)


def test_clustered_resolve_k_reads_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)

    assert resolve_k(
        model,
        OursGcgConfig(),
        cluster_contract=contract,
    ) == (2, 1)


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (OursGcgConfig(top_k_pos=7), (7, 1)),
        (OursGcgConfig(top_k_neg=5), (2, 5)),
        (OursGcgConfig(top_k_pos=7, top_k_neg=5), (7, 5)),
    ],
)
def test_clustered_resolve_k_preserves_override_precedence(
    tmp_path: Path,
    cfg: OursGcgConfig,
    expected: tuple[int, int],
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)

    assert resolve_k(
        model,
        cfg,
        cluster_contract=contract,
    ) == expected


def test_clustered_bucket_layout_still_rejects_more_than_two_buckets(
    tmp_path: Path,
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)
    model.encoder.config.num_buckets = 3
    contract = replace(
        contract,
        selection=replace(contract.selection, num_buckets=3),
    )

    with pytest.raises(
        FeatureGcgError,
        match="require exactly 2 buckets",
    ):
        assert_single_positive_bucket(
            model,
            cluster_contract=contract,
        )


@pytest.mark.parametrize(("surface", "call"), _CONTRACT_CALLS)
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("width", "original SAE width"),
        ("buckets", "bucket count"),
        ("vocab", "in-memory FeatureVocab"),
        ("missing_consistency", "does not expose assert_vocab_consistent"),
        ("failing_consistency", "inconsistent persistent FeatureVocab buffers"),
    ],
)
def test_clustered_loaded_state_mismatch_is_wrapped_with_cause(
    tmp_path: Path,
    surface: str,
    call: Callable[[Any, ClusterFeatureContract], tuple[int, int]],
    case: str,
    match: str,
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)
    if case == "width":
        model.encoder.config.n_sae_features += 1
    elif case == "buckets":
        model.encoder.config.num_buckets += 1
    elif case == "vocab":
        model.feature_vocab = FeatureVocab(4, (0, 3))
    elif case == "missing_consistency":
        del model.assert_vocab_consistent
    elif case == "failing_consistency":

        def _fail_consistency() -> None:
            raise FeatureVocabError("sentinel buffer mismatch")

        model.assert_vocab_consistent = _fail_consistency
    else:
        raise AssertionError(case)

    with pytest.raises(FeatureGcgError, match=match) as exc_info:
        call(model, contract)
    assert surface in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ClusterFeatureContractError)


@pytest.mark.parametrize(
    ("surface", "call", "message"),
    [
        (
            "bucket-layout",
            _bucket_call,
            "feature_gcg bucket-layout checkpoint is missing mapping metadata",
        ),
        (
            "top-k",
            _resolve_call,
            "feature_gcg top-k checkpoint is missing mapping metadata",
        ),
    ],
)
def test_clustered_non_mapping_metadata_raises_feature_gcg_error(
    tmp_path: Path,
    surface: str,
    call: Callable[[Any, ClusterFeatureContract], tuple[int, int]],
    message: str,
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)
    model._loaded_meta_data = []

    with pytest.raises(FeatureGcgError) as exc_info:
        call(model, contract)
    assert surface in str(exc_info.value)
    assert str(exc_info.value) == message


@pytest.mark.parametrize(("surface", "call"), _CONTRACT_CALLS)
def test_cluster_contract_error_is_wrapped_and_cause_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    call: Callable[[Any, ClusterFeatureContract], tuple[int, int]],
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)
    cause = ClusterFeatureContractError(f"{surface} sentinel")

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise cause

    monkeypatch.setattr(
        gcg_config_module,
        "assert_cluster_contract_loaded_state",
        _fail,
    )

    with pytest.raises(FeatureGcgError, match=f"{surface} sentinel") as exc_info:
        call(model, contract)
    assert exc_info.value.__cause__ is cause


@pytest.mark.parametrize(
    ("call", "model", "expected_type", "expected_message"),
    [
        (
            lambda model, **kwargs: assert_single_positive_bucket(model, **kwargs),
            SimpleNamespace(
                encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2))
            ),
            FeatureGcgError,
            "Model lacks _loaded_meta_data attribute. Attach meta_data to the "
            "model after build_encdec_model(): "
            "`model._loaded_meta_data = meta_data`.",
        ),
        (
            lambda model, **kwargs: assert_single_positive_bucket(model, **kwargs),
            SimpleNamespace(_loaded_meta_data={"user_config": {}}),
            FeatureGcgError,
            "Cannot read num_buckets from model.encoder.config: "
            "'types.SimpleNamespace' object has no attribute 'encoder'. "
            "Is this an encoder-decoder model with a loaded encoder?",
        ),
        (
            lambda model, **kwargs: resolve_k(
                model,
                OursGcgConfig(),
                **kwargs,
            ),
            SimpleNamespace(),
            FeatureGcgError,
            "resolve_k: model has no _loaded_meta_data, so top_k for "
            "'feature_selection' cannot be read from the checkpoint's training "
            "selection distribution. Load a checkpoint carrying its selection "
            "metadata, or set OursGcgConfig.top_k_pos / top_k_neg explicitly.",
        ),
        (
            lambda model, **kwargs: resolve_k(
                model,
                OursGcgConfig(),
                **kwargs,
            ),
            SimpleNamespace(_loaded_meta_data={"user_config": {}}),
            FeatureGcgError,
            "resolve_k: checkpoint user_config has no 'feature_selection'; top_k "
            "must come from the training selection distribution. Use a checkpoint "
            "with the selection metadata, or set OursGcgConfig.top_k_pos / "
            "top_k_neg explicitly.",
        ),
    ],
)
def test_none_contract_keeps_error_type_and_message_exact(
    call: Callable[..., tuple[int, int]],
    model: Any,
    expected_type: type[BaseException],
    expected_message: str,
) -> None:
    def _capture(**kwargs: Any) -> BaseException:
        with pytest.raises(BaseException) as exc_info:
            call(model, **kwargs)
        return exc_info.value

    omitted = _capture()
    explicit = _capture(cluster_contract=None)
    for actual in (omitted, explicit):
        assert type(actual) is expected_type
        assert str(actual) == expected_message


@pytest.mark.parametrize(("surface", "call"), _CONTRACT_CALLS)
def test_clustered_gcg_config_validation_is_io_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    call: Callable[[Any, ClusterFeatureContract], tuple[int, int]],
) -> None:
    contract, model, _ = _contract_model_pair(tmp_path)
    real_validator = gcg_config_module.assert_cluster_contract_loaded_state
    validation_calls = 0

    def _validator_spy(*args: Any, **kwargs: Any) -> None:
        nonlocal validation_calls
        validation_calls += 1
        real_validator(*args, **kwargs)

    monkeypatch.setattr(
        gcg_config_module,
        "assert_cluster_contract_loaded_state",
        _validator_spy,
    )
    monkeypatch.setattr(
        t,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"{surface} validation unexpectedly touched disk")
        ),
    )

    call(model, contract)
    assert validation_calls == 1
