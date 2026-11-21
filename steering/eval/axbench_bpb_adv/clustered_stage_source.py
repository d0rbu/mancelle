"""Bind one clustered source validation to its committed run identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from steering.eval.axbench_bpb_adv.artifact_identity import (
    ClusteredSourceIdentity,
    resolve_clustered_source_identity,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceValidation,
)
from steering.eval.cluster_feature_contract import ClusterFeatureContract

_CLUSTERED_STAGE_SOURCE_TOKEN = object()


@dataclass(frozen=True)
class ClusteredStageSource:
    """One exact source snapshot paired with one committed source identity."""

    validation: BpbAdvSourceValidation
    source_identity: ClusteredSourceIdentity
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _CLUSTERED_STAGE_SOURCE_TOKEN:
            raise TypeError(
                "ClusteredStageSource must be built by "
                "resolve_clustered_stage_source()"
            )
        if type(self.validation) is not BpbAdvSourceValidation:
            raise TypeError(
                "clustered stage source requires an exact "
                "BpbAdvSourceValidation"
            )
        if type(self.source_identity) is not ClusteredSourceIdentity:
            raise TypeError(
                "clustered stage source requires an exact "
                "ClusteredSourceIdentity"
            )
        self.validation.__post_init__()
        self.source_identity.__post_init__()
        if (
            self.validation.contract_hash
            != self.source_identity.source_contract_sha256
        ):
            raise AxBenchBpbAdvError(
                "clustered source snapshot and committed source identity disagree"
            )
        if (
            self.validation.contract.feature_contract_fingerprint
            != self.source_identity.feature_contract_fingerprint
        ):
            raise AxBenchBpbAdvError(
                "clustered source snapshot and committed feature identity disagree"
            )


def resolve_clustered_stage_source(
    *,
    validation: BpbAdvSourceValidation,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    expected_source_run_dir: str | Path,
    expected_filter_unaccepted_samples: bool,
) -> ClusteredStageSource:
    """Validate and pair a source snapshot with its committed run identity."""

    if type(validation) is not BpbAdvSourceValidation:
        raise AxBenchBpbAdvError(
            "clustered stage requires an exact BpbAdvSourceValidation"
        )
    validation.__post_init__()
    if (
        validation.contract.schema_version
        != CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION
    ):
        raise AxBenchBpbAdvError(
            "clustered stage requires a schema-v7 source snapshot"
        )
    expected_root = Path(expected_source_run_dir)
    if validation.source_run_dir != expected_root:
        raise AxBenchBpbAdvError(
            "clustered source snapshot belongs to a different source run: "
            f"validated={validation.source_run_dir}, expected={expected_root}"
        )
    if (
        validation.filter_unaccepted_samples
        is not expected_filter_unaccepted_samples
    ):
        raise AxBenchBpbAdvError(
            "clustered source snapshot used a different "
            "filter_unaccepted_samples policy"
        )
    if (
        validation.contract.feature_contract_fingerprint
        != contract.feature_contract_fingerprint
    ):
        raise AxBenchBpbAdvError(
            "clustered source snapshot does not match the target feature contract"
        )
    source_identity = resolve_clustered_source_identity(
        paths=paths,
        contract=contract,
    )
    return ClusteredStageSource(
        validation=validation,
        source_identity=source_identity,
        _token=_CLUSTERED_STAGE_SOURCE_TOKEN,
    )


def require_clustered_stage_source(
    *,
    stage_source: ClusteredStageSource | None,
    contract: ClusterFeatureContract | None,
    surface: str,
) -> ClusteredStageSource | None:
    """Require the paired source value at every clustered consumer boundary."""

    if contract is None:
        if stage_source is not None:
            raise AxBenchBpbAdvError(
                f"unclustered {surface} rejects a clustered stage source"
            )
        return None
    if type(stage_source) is not ClusteredStageSource:
        raise AxBenchBpbAdvError(
            f"clustered {surface} requires one resolved stage source"
        )
    stage_source.__post_init__()
    if (
        stage_source.validation.contract.feature_contract_fingerprint
        != contract.feature_contract_fingerprint
    ):
        raise AxBenchBpbAdvError(
            f"clustered {surface} source does not match the target feature contract"
        )
    return stage_source
