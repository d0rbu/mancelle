"""Shared clustered-evaluation configuration contracts.

The clustered AxBench surfaces deliberately use a narrower schema than the
training pipeline: both the conditioning mask and cluster definition must be
path-backed.  This module also owns the conditional dataclass serialization
needed to keep newly-added ``null`` fields out of legacy run-tag and profiler
payloads.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.conditional.feature_mask import FeatureMaskMode

CLUSTERED_AXBENCH_BASE_SOURCE_METHODS: tuple[str, ...] = (
    "diffmean_decoder_only",
    "neg_diffmean_decoder_only",
    "linear_probe_decoder_only",
    "neg_linear_probe_decoder_only",
    "control_decoder_only",
    "neg_control_decoder_only",
)

CLUSTERED_AXBENCH_BASE_STAGE4_METHODS: tuple[str, ...] = (
    *CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
    "control_encdec",
    "neg_control_encdec",
    "ours_extract_and_add_encdec",
    "ours_replace_random_encdec",
    "ours_replace_all_encdec",
    "neg_ours_remove_encdec",
    "neg_ours_add_to_negatives_encdec",
    "neg_ours_replace_random_negatives_encdec",
    "neg_ours_replace_all_negatives_encdec",
)

# Public AxBench-Base clustered allowlist. Stage-specific validation below the
# config layer keeps the six source producers separate from the Stage 4-only
# encoder-decoder consumers while accepting one explicit end-to-end profile.
CLUSTERED_AXBENCH_BASE_METHODS: tuple[str, ...] = (
    CLUSTERED_AXBENCH_BASE_STAGE4_METHODS
)

CLUSTERED_AXBENCH_BPB_ADV_METHODS: tuple[str, ...] = (
    *CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
    "feature_diffmean_encdec",
    "neg_feature_diffmean_encdec",
    "feature_diffmean_roofline_encdec",
    "neg_feature_diffmean_roofline_encdec",
    "ours_gcg_encdec",
    "neg_ours_gcg_encdec",
    # Same fair bags as ours_gcg_encdec, swept over the class-margin alpha
    # instead of reg_lambda. Encdec-targeted like every other bag-backed method,
    # so the derived ENCDEC set below picks them up without a second listing.
    "ours_gcg_alpha_encdec",
    "neg_ours_gcg_alpha_encdec",
    "ours_gcg_roofline_encdec",
    "neg_ours_gcg_roofline_encdec",
)


#: Clustered methods that run against the encoder-decoder target.
#:
#: Derived, not hand-listed: every clustered method that is not one of the six
#: decoder-only source producers targets the encdec. A drift test cross-checks
#: this against ``Method.target_model`` in the AxBench-Base registry, which is
#: the authority — the constant exists only so this module stays free of a
#: dependency on the registry (and of the import cycle that would create).
#:
#: This is the axis that matters for checkpoint-level guards: the pinned
#: encoder-decoder checkpoint's properties are unreachable from a method whose
#: target is the separate decoder-only model.
CLUSTERED_ENCDEC_TARGET_METHODS: tuple[str, ...] = tuple(
    sorted(
        (
            set(CLUSTERED_AXBENCH_BASE_STAGE4_METHODS)
            | set(CLUSTERED_AXBENCH_BPB_ADV_METHODS)
        )
        - set(CLUSTERED_AXBENCH_BASE_SOURCE_METHODS)
    )
)


#: Methods that never read a checkpoint's auxiliary feature-prediction head.
#:
#: The head (`feature_pred_head.*`, trained under `feature_prediction`) is an
#: auxiliary training-loss output, not part of the conditioning path. The fair
#: and roofline encoder-decoder DiffMean / GCG methods derive their steering
#: signal purely from feature-space conditioning and never consult it, so a
#: checkpoint carrying one is evaluable under exactly these methods.
#:
#: Deliberately NOT a superset of `CLUSTERED_AXBENCH_BPB_ADV_METHODS`: the
#: decoder-only source methods are omitted because whether they may run against
#: a feature-prediction checkpoint is a separate call, not an inference from
#: this one.
CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS: tuple[str, ...] = (
    "feature_diffmean_encdec",
    "neg_feature_diffmean_encdec",
    "feature_diffmean_roofline_encdec",
    "neg_feature_diffmean_roofline_encdec",
    "ours_gcg_encdec",
    "neg_ours_gcg_encdec",
    # The alpha sweep changes only the objective's weighting of the contrast
    # term; it reads the same feature-space conditioning and likewise never
    # consults the feature-prediction head.
    "ours_gcg_alpha_encdec",
    "neg_ours_gcg_alpha_encdec",
    "ours_gcg_roofline_encdec",
    "neg_ours_gcg_roofline_encdec",
)


class ClusterEvalConfigError(ValueError):
    """Raised when a clustered-evaluation config violates its closed schema."""


@dataclass(frozen=True)
class PathFeatureMaskConfig:
    """Path-only evaluation mask config."""

    mode: FeatureMaskMode
    path: str

    def canonical_dict(self) -> dict[str, str]:
        return {"mode": self.mode.value, "path": self.path}


@dataclass(frozen=True)
class PathFeatureClustersConfig:
    """Path-only evaluation cluster config."""

    path: str

    def canonical_dict(self) -> dict[str, str]:
        return {"path": self.path}


def _plain_dataclass_dict(config: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(
            "cluster-aware config serialization requires a dataclass instance, "
            f"got {type(config).__name__}"
        )
    payload = dataclasses.asdict(config)
    if not isinstance(payload, dict):
        raise TypeError("dataclasses.asdict unexpectedly returned a non-mapping")
    return payload


_TRANSPORT_RETRY_BUDGET_KEY = "transport_retry_budget_seconds"
# Identity-preserving defaults for the transport retry budget, frozen as
# literals rather than read from the dataclasses. Reading the live defaults
# would make every future retuning of the budget silently re-map old runs onto
# the new value's tag. ``SynthConfig`` overrides the shared default; every
# other LLM-client section (``judge``, ``concept_examples_qc_judge``, and any
# added later) inherits ``LLMClientConfig``'s. These numbers must never change
# even if the dataclass defaults do.
_LEGACY_SYNTH_RETRY_BUDGET = 300.0
_LEGACY_CLIENT_RETRY_BUDGET = 0.0


def _strip_default_transport_retry_budget(payload: dict[str, Any]) -> None:
    """Drop ``transport_retry_budget_seconds`` from identity at its default.

    The budget is pure transport policy: it decides how long a call waits out
    an upstream 429, never what prompt is sent or what text comes back. It was
    added after these run tags were minted, so leaving it in the payload would
    change the tag of every pre-existing run and orphan its artifacts under a
    new output root — the same hazard the ``feature_clusters`` and
    selection-manifest strips below exist to prevent.

    Every nested LLM-client section is swept rather than a fixed list of two,
    because the config carries a third (``concept_examples_qc_judge``) and
    nothing stops a fourth from being added; a missed section silently
    retags every run.

    A deliberately non-default budget is *not* stripped. Under
    ``qc_fallback_on_exhaust=warn`` a shorter budget drops rows a longer one
    would have kept, so a tuned value can move outputs and has to retag.
    """
    for section, client in payload.items():
        if not isinstance(client, dict) or _TRANSPORT_RETRY_BUDGET_KEY not in client:
            continue
        legacy_default = (
            _LEGACY_SYNTH_RETRY_BUDGET
            if section == "synth"
            else _LEGACY_CLIENT_RETRY_BUDGET
        )
        if client[_TRANSPORT_RETRY_BUDGET_KEY] == legacy_default:
            client.pop(_TRANSPORT_RETRY_BUDGET_KEY, None)


def axbench_base_config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize AxBench-Base while omitting newly introduced disabled fields.

    ``conditioning.feature_mask`` predates clustered evaluation and is never
    removed here. The later ``feature_clusters`` and selection-manifest
    overrides are conditional, preserving every legacy byte when disabled
    while retaining their real values in configured runs.
    """

    payload = _plain_dataclass_dict(config)
    _strip_default_transport_retry_budget(payload)
    if payload.get("force_cluster_shard_revalidation") is False:
        payload.pop("force_cluster_shard_revalidation", None)
    conditioning = payload.get("conditioning")
    if isinstance(conditioning, dict) and conditioning.get("feature_clusters") is None:
        conditioning.pop("feature_clusters", None)
    for key in ("selection_splice_batch_size", "selection_manifest_path"):
        if payload.get(key) is None:
            payload.pop(key, None)
    # #1249: same rule as the keys above -- a config that never fits encdec
    # directions keeps its existing run tag byte-for-byte. When the bag store
    # IS set it participates in identity, which is correct: the fitted encdec
    # directions depend on which committed bags conditioned them.
    if payload.get("encdec_fit_bag_store") is None:
        payload.pop("encdec_fit_bag_store", None)
    return payload


def axbench_bpb_adv_config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize BPB-Adv while omitting newly introduced disabled fields."""

    payload = _plain_dataclass_dict(config)
    # #1249 follow-up: same rule as the stripped keys below -- an empty
    # exclusion list keeps every existing run tag byte-for-byte; a NON-empty
    # one changes the scored population and must change the tag.
    if payload.get("exclude_concept_modes") == []:
        payload.pop("exclude_concept_modes", None)
    # Same rule again: unset means the full default coefficient grid, which is
    # what every existing run swept, so an unset override keeps their tags
    # byte-for-byte. A SET override changes which coefficients are scored and
    # must change the tag.
    if payload.get("steering_coefficient_grid") is None:
        payload.pop("steering_coefficient_grid", None)
    # Same rule one level down, and for the same reason the per-bag alpha stamp
    # is stripped: at the default grid the derive walks exactly the pre-alpha
    # (reg_lambda-only) sequence, so a run that never sweeps alpha must keep its
    # tag byte-for-byte. Left in, this ONE key rehashes every existing run tag.
    ours_gcg = payload.get("ours_gcg")
    if isinstance(ours_gcg, dict) and tuple(
        float(value) for value in ours_gcg.get("objective_alpha_grid", ())
    ) == (DEFAULT_OBJECTIVE_ALPHA,):
        ours_gcg.pop("objective_alpha_grid", None)
    if payload.get("force_cluster_shard_revalidation") is False:
        payload.pop("force_cluster_shard_revalidation", None)
    for key in (
        "feature_mask",
        "feature_clusters",
        "feature_extraction_model_id",
        # Added with the clustered selection-provenance fields. They must be
        # stripped when unset for the same reason as the three above: a config
        # that does not use them has to keep its existing run tag, or every
        # pre-existing run's artifacts are orphaned under a new output root.
        "selection_splice_batch_size",
        "selection_manifest_path",
    ):
        if payload.get(key) is None:
            payload.pop(key, None)
    adversarial_methods = payload.get("adversarial_methods")
    if isinstance(adversarial_methods, list):
        for method in adversarial_methods:
            if not isinstance(method, dict) or method.get("method") != "faster_gcg":
                continue
            # Issue #1227 adds a scoped candidate-only batch size. Its disabled
            # default is byte-neutral; an explicit value survives into run-tag and
            # artifact identity.
            if method.get("candidate_eval_device_batch_size") is None:
                method.pop("candidate_eval_device_batch_size", None)
    return payload


def _strict_nonempty_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClusterEvalConfigError(f"{field} must be a non-empty string")
    return value


def parse_path_feature_mask(raw: Any) -> PathFeatureMaskConfig | None:
    """Parse the clustered-eval path-only mask schema."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ClusterEvalConfigError("feature_mask must be null or a path-only mapping")
    expected = {"mode", "path"}
    keys = set(raw)
    if keys != expected:
        raise ClusterEvalConfigError(
            "clustered evaluation feature_mask must contain exactly "
            f"{sorted(expected)}; missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected, key=str)}"
        )
    try:
        mode = FeatureMaskMode(raw["mode"])
    except (TypeError, ValueError) as exc:
        raise ClusterEvalConfigError(
            "feature_mask.mode must be 'blacklist' or 'whitelist'"
        ) from exc
    return PathFeatureMaskConfig(
        mode=mode,
        path=_strict_nonempty_path(raw["path"], field="feature_mask.path"),
    )


def parse_path_feature_clusters(raw: Any) -> PathFeatureClustersConfig | None:
    """Parse the clustered-eval path-only cluster schema."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ClusterEvalConfigError("feature_clusters must be null or a path-only mapping")
    expected = {"path"}
    keys = set(raw)
    if keys != expected:
        raise ClusterEvalConfigError(
            "feature_clusters must contain exactly one key, 'path'; "
            f"got keys={sorted(keys, key=str)}"
        )
    return PathFeatureClustersConfig(
        path=_strict_nonempty_path(raw["path"], field="feature_clusters.path")
    )


def parse_cluster_eval_paths(
    *,
    feature_mask: Any,
    feature_clusters: Any,
) -> tuple[PathFeatureMaskConfig, PathFeatureClustersConfig] | None:
    """Parse the paired clustered-evaluation config blocks.

    Both blocks are disabled together for a legacy run.  Enabling only one is
    rejected: a cluster partition without the checkpoint mask has ambiguous
    representatives, while a clustered evaluator must never silently treat a
    mask-only run as cluster-safe.
    """

    mask = parse_path_feature_mask(feature_mask)
    clusters = parse_path_feature_clusters(feature_clusters)
    if mask is None and clusters is None:
        return None
    if mask is None or clusters is None:
        missing = "feature_mask" if mask is None else "feature_clusters"
        raise ClusterEvalConfigError(
            "clustered evaluation requires feature_mask and feature_clusters "
            f"together; {missing} is null"
        )
    return mask, clusters


def validate_clustered_method_allowlist(
    methods: list[str] | tuple[str, ...] | None,
    *,
    surface: str,
) -> None:
    """Require an explicit subset of the locked clustered production methods."""

    if methods is None:
        raise ClusterEvalConfigError(
            f"{surface} requires an explicit method allowlist for clustered runs"
        )
    if surface == "axbench_base":
        allowed = CLUSTERED_AXBENCH_BASE_METHODS
    elif surface == "axbench_bpb_adv":
        allowed = CLUSTERED_AXBENCH_BPB_ADV_METHODS
    else:
        raise ClusterEvalConfigError(
            f"unknown clustered method-validation surface {surface!r}"
        )
    unknown = sorted(set(methods) - set(allowed))
    if unknown:
        raise ClusterEvalConfigError(
            f"{surface} clustered method allowlist contains unsupported methods "
            f"{unknown}; allowed={list(allowed)}"
        )
    if not methods:
        raise ClusterEvalConfigError(f"{surface} clustered method allowlist must not be empty")
    if len(methods) != len(set(methods)):
        raise ClusterEvalConfigError(f"{surface} clustered method allowlist contains duplicates")


__all__ = [
    "CLUSTERED_AXBENCH_BASE_METHODS",
    "CLUSTERED_AXBENCH_BASE_SOURCE_METHODS",
    "CLUSTERED_AXBENCH_BASE_STAGE4_METHODS",
    "CLUSTERED_AXBENCH_BPB_ADV_METHODS",
    "ClusterEvalConfigError",
    "PathFeatureClustersConfig",
    "PathFeatureMaskConfig",
    "axbench_base_config_to_dict",
    "axbench_bpb_adv_config_to_dict",
    "parse_cluster_eval_paths",
    "parse_path_feature_clusters",
    "parse_path_feature_mask",
    "validate_clustered_method_allowlist",
]
